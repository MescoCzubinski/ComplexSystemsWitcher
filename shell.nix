{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps:
      with ps; [
        networkx
        matplotlib
        numpy
        pandas
        scipy
        spacy
        spacy-models.en_core_web_sm
      ]))
  ];
}
