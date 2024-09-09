// package com.imperial.joneslab;

// import java.io.BufferedReader;
// import java.io.IOException;
// import java.io.InputStreamReader;


// public class PythonRunner {

//     public static void main(String[] args) {
//         // Step 1: Check for PYTHON_PATH environment variable
//         String pythonPath = System.getenv("PYTHON_PATH");

//         // Step 2: If PYTHON_PATH is not set, default to 'python' in the system path
//         if (pythonPath == null || pythonPath.isEmpty()) {
//             pythonPath = "python";  // Use system default python
//         }

//         try {
//             // Step 3: Use the pythonPath to run the Python script
//             // *** MODIFY the script path if necessary ***
//             ProcessBuilder processBuilder = new ProcessBuilder(pythonPath, "relative/path/to/your_script.py");
//             processBuilder.redirectErrorStream(true);
//             Process process = processBuilder.start();

//             // Step 4: Handle output from the Python process
//             // process.getInputStream().transferTo(System.out); # I think this is a Java 9 feature, try manual instead
//             BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
//             String line;
//             while ((line = reader.readLine()) != null) {
//                 System.out.println(line);  // Print each line to the console
//             }

//             int exitCode = process.waitFor();
//             System.out.println("Python script exited with code: " + exitCode);

//         } catch (IOException | InterruptedException e) {
//             e.printStackTrace();
//         }
//     }
// }

package com.imperial.joneslab;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.File;

public class PythonRunner {

    public static void main(String[] args) {
        // Step 1: Check for PYTHON_PATH environment variable
        String pythonPath = System.getenv("PYTHON_PATH");

        // Step 2: If PYTHON_PATH is not set, default to 'python' in the system path
        if (pythonPath == null || pythonPath.isEmpty()) {
            pythonPath = "python";  // Use system default python
        }

        try {
            // Dynamically locate plugin_dir and microsam_plugin.py
            File currentDir = new File(System.getProperty("user.dir"));  // Get FIJI's current working directory
            File fijiInstallDir = currentDir.getParentFile().getParentFile();  // Navigate to /Applications/Fiji.app
            File pluginDir = new File(fijiInstallDir, "plugin_dir");  // Locate plugin_dir

            // Check if plugin_dir exists
            if (pluginDir.exists() && pluginDir.isDirectory()) {
                File pythonScript = new File(pluginDir, "microsam_plugin.py");  // Locate the Python script

                if (pythonScript.exists()) {
                    // Step 3: Use the pythonPath to run the Python script
                    ProcessBuilder processBuilder = new ProcessBuilder(pythonPath, pythonScript.getAbsolutePath());
                    processBuilder.redirectErrorStream(true);
                    Process process = processBuilder.start();

                    // Step 4: Handle output from the Python process
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);  // Print each line to the console
                    }

                    int exitCode = process.waitFor();
                    System.out.println("Python script exited with code: " + exitCode);
                } else {
                    System.err.println("Python script not found in plugin_dir.");
                }
            } else {
                System.err.println("plugin_dir not found.");
            }

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
