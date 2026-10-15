"""Console binary using the same frozen worker dispatcher as the Windows GUI."""

from jingwei_benchmark.desktop_entry import main

if __name__ == "__main__":
    raise SystemExit(main(desktop=False))
