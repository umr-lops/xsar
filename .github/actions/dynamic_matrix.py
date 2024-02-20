import sys
import urllib.request
import json
import datetime
# minimal python script that returns a strategy matrix, for given github.event_name


with urllib.request.urlopen('https://endoflife.date/api/python.json') as f:
    python_versions = json.loads(f.read().decode('utf-8'))

now = datetime.datetime.now().strftime('%Y-%m-%d')
python_supported_versions = [ v['cycle'] for v in python_versions if v['eol'] > now and v['cycle'] not in ['3.7','3.8','3.12'] ]

python_default_version = python_supported_versions[1]

matrix = {
    'default': {
        'os': ['ubuntu-latest'],
        'python_version': [python_default_version],
    },
    'pull_request': {
        'os': ['ubuntu-latest'],
        'python_version': [python_default_version],
    },
    'schedule': {
        'os': ['ubuntu-latest', 'macos-latest', 'windows-latest'],
        'python_version': python_supported_versions,
    },
    'release': {
        'os': ['ubuntu-latest', 'macos-latest', 'windows-latest'],
        'python_version': python_supported_versions,
    },
    'pull_request_review': {
        'os': ['ubuntu-latest', 'macos-latest', 'windows-latest'],
        'python_version': python_supported_versions,
    },
    'workflow_dispatch': {
        'os': ['ubuntu-latest', 'macos-latest', 'windows-latest'],
        'python_version': python_supported_versions,
    },
}

if __name__ == "__main__":
    try:
        event = sys.argv[1]
    except IndexError:
        event = 'default'
    if event not in matrix:
        event = 'default'

    print('::set-output name=os_matrix::%s' % str(matrix[event]['os']))
    print('::set-output name=python_version_matrix::%s' % str(matrix[event]['python_version']))
