#!/bin/bash

echo '::set-output name=os_matrix::["ubuntu-latest", "macos-latest", "windows-latest"]'
echo '::set-output name=python_version_matrix::["3.7", "3.8", "3.9", "3.10"]'

exit 0

if [ "$1" == "os" ]; then
  echo '["ubuntu-latest", "macos-latest", "windows-latest"]'
fi

if [ "$1" == "python-version" ]; then
  echo '["3.7", "3.8", "3.9", "3.10"]'
fi

