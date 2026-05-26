{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.virtualenv
    pkgs.stdenv.cc.cc.lib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    if [ ! -d .venv ]; then
      echo "Creating virtualenv..."
      virtualenv .venv
      .venv/bin/pip install \
        networkx \
        matplotlib \
        seaborn \
        numpy \
        pandas \
        scipy \
        python-louvain \
        node2vec \
        umap-learn
      echo "Done."
    fi
    source .venv/bin/activate
  '';
}
