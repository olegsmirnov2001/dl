#!/usr/bin/env python3
import typer
import gdown


def download_gdrive_folder(url: str, output_dir: str | None = None) -> None:
    if 'drive.google.com' in url:
        if '/folders/' in url:
            folder_id = url.split('/folders/')[1].split('?')[0]
        else:
            folder_id = url
    else:
        folder_id = url

    folder_url = f'https://drive.google.com/drive/folders/{folder_id}'

    print(f'Downloading folder: {folder_url}')
    gdown.download_folder(folder_url, output=output_dir, quiet=False)
    print('Download complete!')


if __name__ == '__main__':
    typer.run(download_gdrive_folder)
