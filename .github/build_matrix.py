import sys

if __name__ == "__main__":

    print("event: %s" % sys.argv[1])
    print('::set-output name=os_matrix::["ubuntu-latest", "macos-latest", "windows-latest"]')
    print('::set-output name=python_version_matrix::["3.7", "3.8", "3.9", "3.10"]')
