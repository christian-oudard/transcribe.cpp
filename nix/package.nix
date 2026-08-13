# libtranscribe, callable with `pkgs.callPackage ./nix/package.nix {}` for
# non-flake consumers. The flake at the repo root wraps this.
#
# The GPU backend defaults to whatever the platform's is: Metal on Darwin,
# Vulkan elsewhere. Vulkan rather than CUDA because it is in the binary cache,
# needs no unfree toolchain, and covers Intel and AMD as well as NVIDIA. With
# no usable device the library falls back to CPU, so either is safe to build
# on a machine that cannot run it.
{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  shaderc,
  spirv-headers,
  vulkan-headers,
  vulkan-loader,
  apple-sdk_11 ? null,
  # Source and version come from the flake, or from the caller.
  src ? lib.cleanSource ../.,
  version ? "0.2.0",
  withVulkan ? stdenv.hostPlatform.isLinux,
  withMetal ? stdenv.hostPlatform.isDarwin,
  # Off by default: the tests want model files that are not in the tree.
  withTests ? false,
  withTools ? false,
  withExamples ? false,
}:

stdenv.mkDerivation {
  pname = "transcribe-cpp";
  inherit src version;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ] ++ lib.optionals withVulkan [ shaderc ];

  buildInputs =
    lib.optionals withVulkan [
      # ggml-vulkan includes spirv/unified1/spirv.hpp. Nothing passes that
      # include path explicitly: cc-wrapper puts every buildInput on the
      # compiler's search path.
      spirv-headers
      vulkan-headers
      vulkan-loader
    ]
    ++ lib.optionals (withMetal && apple-sdk_11 != null) [ apple-sdk_11 ];

  cmakeFlags = [
    (lib.cmakeBool "TRANSCRIBE_BUILD_SHARED" true)
    (lib.cmakeBool "TRANSCRIBE_VULKAN" withVulkan)
    (lib.cmakeBool "TRANSCRIBE_BUILD_TESTS" withTests)
    (lib.cmakeBool "TRANSCRIBE_BUILD_EXAMPLES" withExamples)
    (lib.cmakeBool "TRANSCRIBE_BUILD_TOOLS" withTools)
    # -march=native would bake the builder's ISA into a store path that
    # claims to be portable, and the binary cache hands it to machines that
    # do not have it.
    (lib.cmakeBool "GGML_NATIVE" false)
  ] ++ lib.optionals withMetal [ (lib.cmakeBool "GGML_METAL" true) ];

  doCheck = withTests;

  meta = {
    description = "Speech recognition in C++ over ggml, covering whisper, parakeet, canary, moonshine and more";
    homepage = "https://github.com/handy-computer/transcribe.cpp";
    license = lib.licenses.mit;
    platforms = lib.platforms.unix;
    mainProgram = "transcribe-cli";
  };
}
