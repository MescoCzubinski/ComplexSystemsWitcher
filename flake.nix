{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ];
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          build-raport = pkgs.writeShellScriptBin "build-raport" ''
            set -e
            cd "$(git rev-parse --show-toplevel)/src/algorithm"
            [ -d figures ] || python solution.py
            pdflatex -interaction=nonstopmode raport.tex
            pdflatex -interaction=nonstopmode raport.tex
          '';
        in {
          default = pkgs.mkShell {
            buildInputs = [
              pkgs.python312
              pkgs.python312Packages.virtualenv
              pkgs.stdenv.cc.cc.lib
              pkgs.miktex
              build-raport
            ];

            shellHook = ''
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH
              if [ ! -d ~/.miktex ]; then
                echo "Setting up MiKTeX..."
                miktexsetup finish
                initexmf --set-config-value="[MPM]AutoInstall=1"
                miktex packages check-update
                miktex packages install cm-super
              fi
              if [ ! -d .venv ]; then
                echo "Creating virtualenv..."
                virtualenv .venv
                .venv/bin/pip install -r requirements.txt
                echo "Done."
              fi
              source .venv/bin/activate
            '';
          };
        });
    };
}
