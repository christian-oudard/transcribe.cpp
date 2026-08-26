{
  description = "Speech recognition in C++ over ggml: whisper, parakeet, canary, moonshine and more";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pkgsFor = system: nixpkgs.legacyPackages.${system};
      version = "0.2.0";
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          build = args: pkgs.callPackage ./nix/package.nix ({ src = self; inherit version; } // args);
        in
        rec {
          default = transcribe-cpp;
          transcribe-cpp = build { };
          # The CLI and the offline tools, for anyone who wants the binaries
          # rather than something to link against.
          transcribe-cpp-tools = build {
            withTools = true;
            withExamples = true;
          };
          # CPU only, for machines with no usable GPU and for reproducing a
          # CPU-path result.
          transcribe-cpp-cpu = build {
            withVulkan = false;
            withMetal = false;
          };
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          default = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.transcribe-cpp ];
            # go for the bindings, uv for the converter and CI gate scripts.
            packages = [
              pkgs.go
              pkgs.uv
            ];
            # The reference environments under scripts/envs are uv-installed
            # manylinux wheels: numpy, torch, NeMo. They dlopen libz and
            # libstdc++ by SONAME and find neither in a nix shell, so
            # `uv run scripts/convert-*.py` dies inside `import numpy` with
            # "libz.so.1: cannot open shared object file". Nothing links
            # against these; they only have to be findable at runtime.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ];
          };
        }
      );

      formatter = forAllSystems (system: (pkgsFor system).nixfmt-tree);
    };
}
