import json
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from common import folders


def download_options(config, output):
    opts = dict(format='bestaudio', outtmpl=str(output / '%(id)s.%(ext)s'),
                postprocessors=[dict(key='FFmpegExtractAudio', preferredcodec='wav')])
    if config['browser']:
        opts['cookiesfrombrowser'] = (config['browser'],)
    return opts


def download_tracks(config, output):
    opts = download_options(config, output)
    tracks = []
    with YoutubeDL(dict(opts, extract_flat='in_playlist', lazy_playlist=True)) as playlist, YoutubeDL(opts) as ydl:
        entries = playlist.extract_info(config['url'], download=False)['entries']
        for entry in entries:
            if not entry:
                continue
            try:
                info = ydl.extract_info(entry['url'], download=False)
                if not info or not 0 < (info.get('duration') or 0) <= config['max_seconds']:
                    continue
                path = output / f"{info['id']}.wav"
                if not path.exists():
                    ydl.process_info(info)
                if path.exists() and path.name not in tracks:
                    tracks.append(path.name)
            except DownloadError as error:
                print(error)
            if len(tracks) == config['count']:
                break
    return tracks


def main():
    source, output = folders(__file__)
    config = json.loads((source / 'config.json').read_text())
    tracks = download_tracks(config, output)
    (output / 'tracks.json').write_text(json.dumps(tracks, indent=2))
    print(f'{len(tracks)}/{config["count"]} eligible tracks downloaded')


if __name__ == '__main__':
    main()
