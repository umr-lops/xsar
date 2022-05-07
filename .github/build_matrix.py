import sys

matrix = {
    'default': {
        'os': ['ubuntu-latest'],
        'python_version': ['3.10']
    }

}

if __name__ == "__main__":
    event = sys.argv[1]
    if event not in matrix:
        event = 'default'


    print('::set-output name=os_matrix::%s' % str(matrix[event]['os']))
    print('::set-output name=python_version_matrix::%s' % str(matrix[event]['python_version']))
