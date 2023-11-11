# --------------------------------------------------------------------------------------------------------------------
# This script collects responses of Systemfailure Deepfake Detection HTTP Service for local directory with video files
# --------------------------------------------------------------------------------------------------------------------

import os
import sys
import json
import aiohttp
import asyncio
from tqdm import tqdm
from argparse import ArgumentParser
from apiutils import scan_all_sub_folders, request_liveness, guess_mime_type


argparser = ArgumentParser('antideepfakes local directory checker')
argparser.add_argument('--base_url', default='http://localhost:5000', help='API base url')
argparser.add_argument('--path_prefix', default='face', help='path prefix')
argparser.add_argument('--local_path', default=None, help='local directory path to check')
argparser.add_argument('--max_parallel_requests', default=1, type=int, help='self explained')
argparser.add_argument('--out', default='./antideepfakes.csv', help='filename where to save result')
args = argparser.parse_args()


async def main():
    client_session = aiohttp.ClientSession(args.base_url)

    files_list = []
    for folder in scan_all_sub_folders(args.local_path, args.local_path):
        files_list += [os.path.join(folder, f.name) for f in os.scandir(folder) if f.is_file()
                       and guess_mime_type(f.name) is not None]
    files_total = len(files_list)

    with open(args.out, 'w') as csv:
        csv.write("FILENAME, HTTP_STATUS, JSON_REPLY")
        progress_bar = tqdm(total=files_total, file=sys.stdout)
        tasks_finished = 0
        while tasks_finished < files_total:
            last = files_total - tasks_finished
            calls = args.max_parallel_requests if last > args.max_parallel_requests else last
            tasks = []
            for i in range(calls):
                filename = files_list[tasks_finished + i]
                with open(filename, 'rb') as i_f:
                    bin_data = i_f.read()
                tasks.append(asyncio.create_task(request_liveness(client_session, args.path_prefix, bin_data, filename)))
            if len(tasks) > 0:
                await asyncio.gather(*tasks)
                for i in range(calls):
                    http_status, json_reply = tasks[i].result()
                    line = f"\n{files_list[tasks_finished + i]}, {http_status}, " \
                           f"{json.dumps(json_reply, separators=(',',':'), ensure_ascii=False)}"
                    csv.write(line)
                    csv.flush()
            progress_bar.update(calls)
            tasks_finished += calls

        progress_bar.close()

    await client_session.close()


if __name__ == '__main__':
    if not os.path.exists(args.local_path):
        print(f"Local path {args.local_path} does not exist! Abort...")
        exit(1)
    target_path = args.out.rsplit('/', 1)[0]
    if not os.path.exists(target_path):
        print(f"Target path {target_path} does not exist! Abort...")
        exit(2)
    asyncio.run(main())
