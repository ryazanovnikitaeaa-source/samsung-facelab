"""Start on localhost, open browser, leave console available to stop."""
import argparse
from threading import Timer
import webbrowser

import uvicorn

from facelab.web import create_app


def main():
    parser=argparse.ArgumentParser(description='FaceLab local application')
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    url=f'http://127.0.0.1:{args.port}'
    print(f'FaceLab: {url}\nStop: Ctrl+C',flush=True)
    if not args.no_browser:
        timer=Timer(1.5,lambda:webbrowser.open(url)); timer.daemon=True; timer.start()
    uvicorn.run(create_app(),host='127.0.0.1',port=args.port,log_level='info')


if __name__=='__main__':
    main()
