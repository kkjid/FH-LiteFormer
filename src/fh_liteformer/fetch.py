"""Download, merge, hash-check and safely extract the registered data/model archives."""
import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, quote, urlparse
from urllib.request import urlopen
from .io import load_json, sha256


def safe_destination(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes the output directory')
    return target


def obtain(base, name, path, expected):
    if Path(name).name != name:
        raise ValueError('Release asset name must be a plain filename')
    if path.exists():
        if sha256(path) == expected:
            return
        raise ValueError(f'Existing download has a different hash: {path.name}')
    temporary = path.with_suffix(path.suffix + '.partial')
    local = Path(base)
    if local.is_dir():
        with (local / name).open('rb') as source, temporary.open('wb') as destination:
            shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
    else:
        if urlparse(base).scheme not in ('http', 'https'):
            raise ValueError('base-url must be an existing local directory or HTTP(S) URL')
        with urlopen(urljoin(base.rstrip('/') + '/', quote(name)), timeout=60) as source, temporary.open('wb') as destination:
            shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
    if sha256(temporary) != expected:
        raise ValueError(f'Download SHA-256 mismatch: {name}')
    temporary.replace(path)


def install_group(group, base, output, cache):
    merged = cache / (group['name'] + '.tar')
    if not merged.exists():
        temporary = merged.with_suffix('.merging')
        with temporary.open('wb') as destination:
            for part in group['parts']:
                path = cache / part['name']
                obtain(base, part['name'], path, part['sha256'])
                if path.stat().st_size != part['bytes']:
                    raise ValueError('Release part size mismatch')
                with path.open('rb') as source:
                    shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
        if sha256(temporary) != group['archive_sha256']:
            raise ValueError('Merged archive SHA-256 mismatch')
        temporary.replace(merged)
    elif sha256(merged) != group['archive_sha256']:
        raise ValueError('Cached merged archive hash mismatch')
    prefixes = group['prefixes']
    with tarfile.open(merged, 'r:') as archive:
        for member in archive:
            if not member.isfile() or not any(member.name.startswith(prefix + '/') for prefix in prefixes):
                raise ValueError(f'Unregistered archive entry: {member.name}')
            target = safe_destination(output, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError('Unreadable archive entry')
            if target.exists():
                digest = hashlib.sha256()
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(block)
                if sha256(target) != digest.hexdigest():
                    raise ValueError(f'Extraction would overwrite different bytes: {member.name}')
            else:
                with target.open('xb') as destination:
                    shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
            source.close()
    print(f'Installed and verified archive: {group["name"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('release-manifest.json'))
    parser.add_argument('--base-url', required=True, help='Local asset directory or an actual published Release assets URL')
    parser.add_argument('--output', type=Path, default=Path('.'))
    parser.add_argument('--groups', nargs='+', choices=['processed', 'weights', 'training', 'supplement'], default=['processed', 'weights'])
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / 'downloads'
    cache.mkdir(exist_ok=True)
    for name in args.groups:
        install_group(manifest['groups'][name], args.base_url, args.output, cache)


if __name__ == '__main__':
    main()
