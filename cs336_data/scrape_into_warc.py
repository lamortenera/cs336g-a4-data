# /// script
# dependencies = [
#     "aiohttp",
#     "warcio",        
#     "asyncio",        
# ]
# ///

import asyncio
import aiohttp
import threading
from warcio.warcwriter import WARCWriter
from warcio.statusandheaders import StatusAndHeaders
import argparse
import io
import sys

parser = argparse.ArgumentParser()
parser.add_argument("input", help="the input list of URLs")
parser.add_argument("output", help="the output WARC file")
parser.add_argument("--num_threads", help="the number of threads to use", type=int, default=10)
parser.add_argument("--timeout", help="the timeout for each request", type=int, default=5)

# Fake browser headers to bypass bot blocks and 202 Status codes
CUSTOM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate"
}

warc_write_lock = threading.Lock()

async def fetch_and_archive(session, url, writer, semaphore, timeout):
    async with semaphore:
        try:
            async with session.get(url, timeout=timeout, headers=CUSTOM_HEADERS) as response:
                if response.status != 200:
                    print("X", end='')
                    return
                else:
                    print("|", end='')
                html_content = await response.read()
                payload_stream = io.BytesIO(html_content)

                
                # Format headers for the WARC file
                headers_list = [(k, v) for k, v in response.headers.items()]
                http_headers = StatusAndHeaders(f"{response.status} {response.reason}", headers_list, protocol='HTTP/1.1')
                
                # Write directly to the WARC file
                record = writer.create_warc_record(url, 'response', http_headers=http_headers, payload=payload_stream)
                
                with warc_write_lock:
                    writer.write_record(record)
        except asyncio.TimeoutError:
            print("T", end='')
        except Exception as e:
            print("E", end='')
            # Print to standard error a more detailed error message
            # print(f"Error fetching {url}: {e}", file=sys.stderr)


async def main():
    args = parser.parse_args()
    semaphore = asyncio.Semaphore(args.num_threads)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    
    with open(args.input, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    with open(args.output, 'wb') as warc_file:
        writer = WARCWriter(warc_file, gzip=True)
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_and_archive(session, url, writer, semaphore, timeout) for url in urls]
            await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
