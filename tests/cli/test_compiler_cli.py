import subprocess
import unittest
import os

class CompilerCliTest(unittest.TestCase):

    def setUp(self):
        self.compiler_path = os.path.join(os.environ.get("GLOW_BIN_DIR", "bin"), "glow-compiler")
        # Create a dummy model file for testing
        with open("dummy_model.onnx", "w") as f:
            f.write("dummy onnx model")

    def tearDown(self):
        if os.path.exists("dummy_model.onnx"):
            os.remove("dummy_model.onnx")
        if os.path.exists("dummy_model.o"):
            os.remove("dummy_model.o")

    def test_compiler_help(self):
        """Test the compiler's help message."""
        result = subprocess.run([self.compiler_path, "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage: glow-compiler", result.stdout)

    def test_compiler_version(self):
        """Test the compiler's version output."""
        result = subprocess.run([self.compiler_path, "--version"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertRegex(result.stdout, r"Glow compiler version \d+\.\d+\.\d+")

    def test_compile_model(self):
        """Test compiling a dummy model."""
        result = subprocess.run([self.compiler_path, "dummy_model.onnx", "-o", "dummy_model.o"], capture_output=True, text=True)
        # This will fail if the dummy model is not a real model, so we check for a specific error message
        # In a real test, we would use a valid model and check for success
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error: Invalid model format.", result.stderr)

if __name__ == "__main__":
    unittest.main()
