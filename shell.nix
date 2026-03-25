{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps:
      with ps; [
        networkx
        matplotlib
        scipy
        numpy
        pandas
        spacy
        spacy-models.en_core_web_sm
      ]))
  ];
}
