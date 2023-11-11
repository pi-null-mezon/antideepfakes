import os
import aiohttp
import json


def scan_all_sub_folders(path, remove_prefix):
    sub_folders = []
    current_level = [s.name for s in os.scandir(path) if s.is_dir()]
    sub_folders.append(path)
    if len(current_level) > 0:
        for sub_folder_name in current_level:
            sub_folders += scan_all_sub_folders(os.path.join(path, sub_folder_name), remove_prefix)
    return sorted(sub_folders)


def guess_mime_type(filename: str):
    parts = filename.rsplit('.', 1)
    if len(parts) == 1:
        return None
    ext = parts[1].lower()
    if ext == 'mp4':
        return 'video/mp4'
    elif ext == 'webm':
        return 'video/webm'
    elif ext == 'mov':
        return 'video/quicktime'
    elif ext == 'avi':
        return 'video/x-msvideo'
    return None


async def request_liveness(session, path_prefix, bio_sample, filename):
    data = aiohttp.FormData()
    mime_type = guess_mime_type(filename)
    assert mime_type is not None
    data.add_field('bio_sample', bio_sample, filename=filename, content_type=mime_type)
    metadata = b'{"mnemonic": "passive-instructions", "actions": [{"type": "deepfake-type"}]}'
    data.add_field('metadata', metadata, content_type="application/json")
    endpoint = f'/v1/{path_prefix}/liveness/detect' if path_prefix != '' else '/v1/liveness/detect'
    try:
        async with session.post(endpoint, data=data) as response:
            future = await response.read()
            if response.status in [200, 400, 500]:
                try:
                    future_json = json.loads(future.decode('utf-8'))
                except json.decoder.JSONDecodeError:
                    print(f"{[response.status]} - {future.decode('utf-8')}")
                    future_json = {}
            else:
                print(f"{[response.status]} - {future.decode('utf-8')}")
                future_json = {}
            return response.status, future_json
    except aiohttp.client_exceptions.ClientConnectorError as ex:
        print("Can not connect to server! Check paths!")
    except aiohttp.client_exceptions.ClientOSError as ex:
        print("Can not write to endpoint! Check paths!")
    return 0, {}
