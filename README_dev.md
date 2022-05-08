# Developement notes

## Making a new conda release

### When ?

The [conda-feedstok workflow](https://github.com/umr-lops/xsar/actions/workflows/conda-feedstock-check.yml) is scheduled once a week.
It's main purpose is to check that the documentation from the dev branch can be generated with the conda package.

If not, an error will be raised, indicating that's we need to build a new conda release.


### How ?

Conda package can only be built if a release tag has been set

  * Go to [new release](https://github.com/umr-lops/xsar/releases/new) and choose a tag name like `v0.9`
  * Download the `.tar.gz` you've just created, and get `sha256`
    ```
    curl -sL https://github.com/umr-lops/xsar/archive/refs/tags/v0.9.1.tar.gz | openssl sha256
    ```
  * fork https://github.com/conda-forge/xsar-feedstock
  * In the forked repository, edit `main/recipe/meta.yaml`
  * Change `{% set version = "0.7" %}` to the new tag you've just created (without the 'v')
  * Change `sha256: a68663...fc1c28` to the `sha256` you've computed above
  * Check needed dependancies

    The `run` section should be a copy/paste from `dependancies` section in [environment.yml](https://github.com/umr-lops/xsar/blob/develop/environment.yml)
  * Submit the pull request, and follow instruction 
    
    See all checkboxes, and don't forget to add a comment `@conda-forge-admin, please rerender`
  * If possible, wait for a reviewer for comments
  * Merge the pull request
  * After a while (30m-2h), you should
    * See your new versio under https://github.com/conda-forge/xsar-feedstock
    * Find the new version with
    ```
    conda search -c conda-forge xsar
    ```
  * Manually trigger workflow https://github.com/umr-lops/xsar/actions/workflows/conda-feedstock-check.yml
    If everything is correct, the workflow should success. 
    Otherwise, it seems that there is a problem with the package. 
    Fix it, and go to step 0, increasing the version number by 0.0.1..
  
    
