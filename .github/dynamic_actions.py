import sys
# minimal python script that returns a strategy matrix, for given github.event_name

matrix = {
    'default': {
        'os': ['ubuntu-latest'],
        'python_version': ['3.10'],
        'doc': False
    },
    'pull_request': {
        'os': ['ubuntu-latest'],
        'python_version': ['3.10'],
        'doc': False
    },
    'schedule': {
        'os': ['ubuntu-latest', 'macos-latest', 'windows-latest'],
        'python_version': ['3.7', '3.8', '3.9', '3.10'],
        'doc': True
    },

}

if __name__ == "__main__":
    event = sys.argv[1]
    if event not in matrix:
        event = 'default'

    print('::set-output name=os_matrix::%s' % str(matrix[event]['os']))
    print('::set-output name=python_version_matrix::%s' % str(matrix[event]['python_version']))
    print('::set-output name=doc::%s' % str(matrix[event]['doc']))
