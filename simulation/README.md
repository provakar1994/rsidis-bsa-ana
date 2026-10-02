# Local simulation checkout

`simc_gfortran/` is an independent Git repository, ignored by the parent SSA
repository. Its configured source is
https://github.com/provakar1994/simc_gfortran.git.

From this directory, a fresh local checkout can be created with:

```sh
git clone https://github.com/provakar1994/simc_gfortran.git
```

Manage simulation changes and commits inside that checkout. It is not a
submodule of the SSA repository.
