// package com.imperial.joneslab;

// import java.awt.GraphicsEnvironment;
// import org.scijava.command.Command;
// import org.scijava.plugin.Parameter;
// import org.scijava.plugin.Plugin;
// import org.scijava.ui.UIService;

// import net.imagej.ImageJ;
// import net.imagej.ops.OpService;
// import net.imglib2.type.numeric.RealType;

// import py4j.GatewayServer;
// import java.io.BufferedReader;
// import java.io.InputStreamReader;

// import java.io.FileWriter;
// import java.io.IOException;
// import java.io.PrintWriter;
// import java.lang.management.ManagementFactory;
// import java.lang.management.ThreadMXBean;
// import java.util.Date;

// import java.io.IOException;
// import java.io.File;

// @Plugin(type = Command.class, menuPath = "Plugins>Jones Lab Segmentation")
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
//             ProcessBuilder processBuilder = new ProcessBuilder(pythonPath, "path/to/your_script.py");
//             processBuilder.redirectErrorStream(true);
//             Process process = processBuilder.start();

//             // Step 4: Handle output from the Python process
//             process.getInputStream().transferTo(System.out);
//             int exitCode = process.waitFor();
//             System.out.println("Python script exited with code: " + exitCode);

//         } catch (IOException | InterruptedException e) {
//             e.printStackTrace();
//         }
//     }
// }


// public class MlSegmentation<T extends RealType<T>> implements Command {

//     @Parameter
//     private UIService uiService;

//     @Parameter
//     private OpService opService;

//     @Override
//     public void run() {
//         try {
//             // Check if we are in a headless environment
//             if (GraphicsEnvironment.isHeadless()) {
//                 System.out.println("Running in headless mode");
//                 // If headless, run the code without UI
//                 runHeadless();
//             } else {
//                 System.out.println("Running with GUI");
//                 // If not headless, run the code with UI
//                 runWithUI();
//             }
//         } catch (Exception e) {
//             e.printStackTrace();
//         }
//     }

//     public static void main(final String... args) throws Exception {
//         new MlSegmentation<>().run();
//     }

//     /******************************************* HEADLESS FUNCTIONALITY *******************************************/
//     public static void runHeadless() throws Exception {
//         System.out.println("Hello from headless mode!");
//         // Implement headless functionality here
//     }

//     /******************************************* INTERACTIVE FUNCTIONALITY *******************************************/
//     public static void runWithUI() throws Exception {
//         ImageJ ij = ImageJSingleton.getInstance();  // Correctly assign the singleton instance to the variable
//         GatewayServer server = null; // Define server outside the try block
//         Process process = null; // Define process outside the try block

//         try {
//             System.out.println("Running plugin in interactive mode");

//             // Set up Py4J gateway
//             PythonGateway pyGateway = new PythonGateway();
//             server = new GatewayServer(pyGateway, 0);
//             server.start();
//             System.out.println("Gateway server started");

//             // Variable to instruct Java to use micro-sam virtual environment
//             String pythonInterpreter = "/vol/biomedic3/bglocker/mscproj24/nma23/miniforge3/envs/micro-sam/bin/python";

//             // Execute Python script to set up the Python Gateway
//             ProcessBuilder processBuilder = new ProcessBuilder(pythonInterpreter, "/vol/biomedic3/bglocker/mscproj24/nma23/nma23_code/microsam_plugin.py");
//             processBuilder.redirectErrorStream(true);
//             process = processBuilder.start();
//             System.out.println("Started Python process");

//             monitorThreadCount(process);
//             System.out.println("Monitoring thread count");

//             // Debugging: print Python outputs to console
//             BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
//             String line;
//             while ((line = reader.readLine()) != null) {
//                 System.out.println(line);
//             }

//             // Check if Python script executed
//             int exitCode = process.waitFor();
//             if (exitCode != 0) {
//                 System.err.println("Python script exited with error code: " + exitCode);
//             }
            
//         } catch (Exception e) {
//             System.err.println("Exception occurred during runWithUI execution: " + e.getMessage());
//             e.printStackTrace();
//         } finally {
//             // Ensure that resources are properly cleaned up
//             if (server != null) {
//                 System.out.println("Shutting down GatewayServer");
//                 server.shutdown();
//             }

//             if (process != null) {
//                 System.out.println("Destroying Python process");
//                 process.destroy();
//             }

//             if (ij != null && ij.context() != null) {
//                 System.out.println("Shutting down ImageJ");
//                 ij.context().dispose();
//             }

//             if (ij != null && ij.ui() != null) {
//                 System.out.println("Disposing ImageJ UI");
//                 ij.ui().dispose();
//             }
//         }

//         System.out.println("Exiting runWithUI() method");
//     }

//     // Class to handle communication with Python
//     public static class PythonGateway {
//         public void invokePythonFunction() {
//             // Code to call Python function via Py4J
//             System.out.println("This text is being printed from invokePythonFunction()");
//         }
//     }

//     // Monitor the number of threads and kill the process if it exceeds a certain count
//     private static void monitorThreadCount(Process process) {
//         new Thread(() -> {
//             boolean processKilled = false;
//             try (PrintWriter logWriter = new PrintWriter(new FileWriter("process_kill_log.txt", true))) {
//                 while (true) {
//                     int threadCount = getThreadCount();
//                     System.out.println("Current thread count: " + threadCount);
//                     if (threadCount > 40 && !processKilled) {
//                         String errorMessage = "Thread count exceeded 40 at " + new Date() + ". Killing the process.";
//                         System.err.println(errorMessage);
//                         logWriter.println(errorMessage);
//                         process.destroy();
//                         logWriter.println("Process killed at " + new Date());
//                         processKilled = true;
//                         break;
//                     }
//                     Thread.sleep(1000); // Check every second
//                 }
//             } catch (InterruptedException | IOException e) {
//                 e.printStackTrace();
//             }
//         }).start();
//     }

//     // Get the current thread count
//     private static int getThreadCount() {
//         ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
//         return threadMXBean.getThreadCount();
//     }
// }

// // Singleton Class for ImageJ
// class ImageJSingleton {
//     private static ImageJ instance;

//     private ImageJSingleton() {}

//     public static synchronized ImageJ getInstance() {
//         if (instance == null) {
//             instance = new ImageJ();
//             System.out.println("Created new ImageJ instance");
//         } else {
//             System.out.println("Using existing ImageJ instance");
//         }
//         return instance;
//     }
// }

// ##################################################################################3
// Need to export PYTHON_PATH AND PYTHON_SCRIPT_PATH as environment variables. Run:

// (for mac, then launch FIJI from terminal)
// export PYTHON_PATH=/usr/bin/python3
// export PYTHON_SCRIPT_PATH=/Users/nicoleaudia/microsam_plugin.py
// cd /Applications/Fiji.app/Contents/MacOS
// ./ImageJ-macosx

// (or for windows)
// set PYTHON_PATH=C:\Python39\python.exe
// set PYTHON_SCRIPT_PATH=C:\path\to\microsam_plugin.py
// (then launch FIJI from terminal)









//////////// TAKE 2




// package com.imperial.joneslab;

// import java.awt.GraphicsEnvironment;
// import org.scijava.command.Command;
// import org.scijava.plugin.Parameter;
// import org.scijava.plugin.Plugin;
// import org.scijava.ui.UIService;
// import net.imagej.ImageJ;
// import net.imagej.ops.OpService;
// import net.imglib2.type.numeric.RealType;
// import py4j.GatewayServer;
// import java.io.BufferedReader;
// import java.io.InputStreamReader;
// import java.io.FileWriter;
// import java.io.IOException;
// import java.io.PrintWriter;
// import java.lang.management.ManagementFactory;
// import java.lang.management.ThreadMXBean;
// import java.util.Date;
// import java.io.IOException;
// import java.io.File;

// @Plugin(type = Command.class, menuPath = "Plugins>Jones Lab Segmentation")


// public class MlSegmentation<T extends RealType<T>> implements Command {

//     @Parameter
//     private UIService uiService;

//     @Parameter
//     private OpService opService;

//     @Override
//     public void run() {
//         try {
//             // Check if we are in a headless environment
//             if (GraphicsEnvironment.isHeadless()) {
//                 System.out.println("Running in headless mode");
//                 runHeadless();  // Run headless mode logic
//             } else {
//                 System.out.println("Running with GUI");
//                 runWithUI();  // Run interactive mode logic
//             }
//         } catch (Exception e) {
//             e.printStackTrace();
//         }
//     }

//     public static void main(final String... args) throws Exception {
//         new MlSegmentation<>().run();
//     }

//     /******************************************* HEADLESS FUNCTIONALITY *******************************************/
//     public static void runHeadless() throws Exception {
//         System.out.println("Hello from headless mode!");
//         // Implement headless functionality here if needed
//     }

//     /******************************************* INTERACTIVE FUNCTIONALITY *******************************************/
//     public static void runWithUI() throws Exception {
//         ImageJ ij = ImageJSingleton.getInstance();
//         GatewayServer server = null;
//         Process process = null;

//         try {
//             System.out.println("Running plugin in interactive mode");

//             // Set up Py4J gateway
//             PythonGateway pyGateway = new PythonGateway();
//             server = new GatewayServer(pyGateway, 0);
//             server.start();
//             System.out.println("Gateway server started");

//             // Step 1: Dynamically load Python interpreter from environment variable
//             String pythonInterpreter = System.getenv("PYTHON_PATH");
//             if (pythonInterpreter == null || pythonInterpreter.isEmpty()) {
//                 pythonInterpreter = "python";  // Default to system python
//             }

//             // Step 2: Dynamically load Python script path from environment variable
//             // *** MODIFY the script path if necessary ***
//             String pythonScriptPath = System.getenv("PYTHON_SCRIPT_PATH");
//             if (pythonScriptPath == null || pythonScriptPath.isEmpty()) {
//                 pythonScriptPath = "relative/path/to/microsam_plugin.py";  // Use relative or default path
//             }

//             // Execute Python script to set up the Python Gateway
//             ProcessBuilder processBuilder = new ProcessBuilder(pythonInterpreter, pythonScriptPath);
//             processBuilder.redirectErrorStream(true);
//             process = processBuilder.start();
//             System.out.println("Started Python process");

//             monitorThreadCount(process);

//             // Debugging: print Python outputs to console
//             BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
//             String line;
//             while ((line = reader.readLine()) != null) {
//                 System.out.println(line);
//             }

//             // Check if Python script executed correctly
//             int exitCode = process.waitFor();
//             if (exitCode != 0) {
//                 System.err.println("Python script exited with error code: " + exitCode);
//             }

//         } catch (Exception e) {
//             System.err.println("Exception occurred during runWithUI execution: " + e.getMessage());
//             e.printStackTrace();
//         } finally {
//             if (server != null) {
//                 server.shutdown();
//             }
//             if (process != null) {
//                 process.destroy();
//             }
//             if (ij != null && ij.context() != null) {
//                 ij.context().dispose();
//             }
//             if (ij != null && ij.ui() != null) {
//                 ij.ui().dispose();
//             }
//         }

//         System.out.println("Exiting runWithUI() method");
//     }

//     // Class to handle communication with Python via Py4J
//     public static class PythonGateway {
//         public void invokePythonFunction() {
//             System.out.println("This text is being printed from invokePythonFunction()");
//         }
//     }

//     // Monitor the number of threads and kill the process if it exceeds a certain count
//     private static void monitorThreadCount(Process process) {
//         new Thread(() -> {
//             boolean processKilled = false;
//             try (PrintWriter logWriter = new PrintWriter(new FileWriter("process_kill_log.txt", true))) {
//                 // Dynamically load MAX_THREAD_COUNT environment variable
//                 int maxThreadCount = 60;  // Default value
//                 String threadCountEnv = System.getenv("MAX_THREAD_COUNT");
//                 if (threadCountEnv != null && !threadCountEnv.isEmpty()) {
//                     maxThreadCount = Integer.parseInt(threadCountEnv);  // Load max thread count from environment
//                 }

//                 while (true) {
//                     int threadCount = getThreadCount();
//                     System.out.println("Current thread count: " + threadCount);
//                     if (threadCount > maxThreadCount && !processKilled) {
//                         String errorMessage = "Thread count exceeded " + maxThreadCount + " at " + new Date() + ". Killing the process.";
//                         System.err.println(errorMessage);
//                         logWriter.println(errorMessage);
//                         process.destroy();
//                         logWriter.println("Process killed at " + new Date());
//                         processKilled = true;
//                         break;
//                     }
//                     Thread.sleep(1000);  // Check thread count every second
//                 }
//             } catch (InterruptedException | IOException e) {
//                 e.printStackTrace();
//             }
//         }).start();
//     }

//     // Get the current thread count
//     private static int getThreadCount() {
//         ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
//         return threadMXBean.getThreadCount();
//     }
// }

// // Singleton Class for ImageJ instance
// class ImageJSingleton {
//     private static ImageJ instance;

//     private ImageJSingleton() {}

//     public static synchronized ImageJ getInstance() {
//         if (instance == null) {
//             instance = new ImageJ();
//             System.out.println("Created new ImageJ instance");
//         } else {
//             System.out.println("Using existing ImageJ instance");
//         }
//         return instance;
//     }
// }


package com.imperial.joneslab;

import java.awt.GraphicsEnvironment;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.type.numeric.RealType;
import py4j.GatewayServer;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Date;
import java.io.File;

@Plugin(type = Command.class, menuPath = "Plugins>Jones Lab Segmentation")
public class MlSegmentation<T extends RealType<T>> implements Command {

    @Parameter
    private UIService uiService;

    @Parameter
    private OpService opService;

    @Override
    public void run() {
        try {
            if (GraphicsEnvironment.isHeadless()) {
                System.out.println("Running in headless mode");
                runHeadless();
            } else {
                System.out.println("Running with GUI");
                runWithUI();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(final String... args) throws Exception {
        new MlSegmentation<>().run();
    }

    public static void runHeadless() throws Exception {
        System.out.println("Hello from headless mode!");
    }

    public static void runWithUI() throws Exception {
        ImageJ ij = ImageJSingleton.getInstance();
        GatewayServer server = null;
        Process process = null;

        try {
            System.out.println("Running plugin in interactive mode");

            // Set up Py4J gateway
            PythonGateway pyGateway = new PythonGateway();
            server = new GatewayServer(pyGateway, 0);
            server.start();
            System.out.println("Gateway server started");

            // Dynamically locate plugin_dir and microsam_plugin.py
            File currentDir = new File(System.getProperty("user.dir"));  // Get FIJI's current working directory
            File fijiInstallDir = currentDir.getParentFile().getParentFile();  // Navigate to /Applications/Fiji.app
            File pluginDir = new File(fijiInstallDir, "plugin_dir");  // Locate plugin_dir

            // Check if plugin_dir exists
            if (pluginDir.exists() && pluginDir.isDirectory()) {
                File pythonScript = new File(pluginDir, "microsam_plugin.py");  // Locate the Python script

                if (pythonScript.exists()) {
                    // Step 1: Dynamically load Python interpreter from environment variable
                    String pythonInterpreter = System.getenv("PYTHON_PATH");
                    if (pythonInterpreter == null || pythonInterpreter.isEmpty()) {
                        pythonInterpreter = "python";  // Default to system python
                    }

                    // Execute Python script
                    ProcessBuilder processBuilder = new ProcessBuilder(pythonInterpreter, pythonScript.getAbsolutePath());
                    processBuilder.redirectErrorStream(true);
                    process = processBuilder.start();
                    System.out.println("Started Python process");

                    monitorThreadCount(process);

                    // Print Python outputs to console
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    // Check if Python script executed successfully
                    int exitCode = process.waitFor();
                    if (exitCode != 0) {
                        System.err.println("Python script exited with error code: " + exitCode);
                    }

                } else {
                    System.err.println("Python script not found in plugin_dir.");
                }
            } else {
                System.err.println("plugin_dir not found.");
            }

        } catch (Exception e) {
            System.err.println("Exception occurred during runWithUI execution: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (server != null) {
                server.shutdown();
            }
            if (process != null) {
                process.destroy();
            }
            if (ij != null && ij.context() != null) {
                ij.context().dispose();
            }
            if (ij != null && ij.ui() != null) {
                ij.ui().dispose();
            }
        }

        System.out.println("Exiting runWithUI() method");
    }

    public static class PythonGateway {
        public void invokePythonFunction() {
            System.out.println("This text is being printed from invokePythonFunction()");
        }
    }

    private static void monitorThreadCount(Process process) {
        new Thread(() -> {
            boolean processKilled = false;
            try (PrintWriter logWriter = new PrintWriter(new FileWriter("process_kill_log.txt", true))) {
                while (true) {
                    int threadCount = getThreadCount();
                    System.out.println("Current thread count: " + threadCount);
                    if (threadCount > 60 && !processKilled) {
                        String errorMessage = "Thread count exceeded 60 at " + new Date() + ". Killing the process.";
                        System.err.println(errorMessage);
                        logWriter.println(errorMessage);
                        process.destroy();
                        logWriter.println("Process killed at " + new Date());
                        processKilled = true;
                        break;
                    }
                    Thread.sleep(1000);
                }
            } catch (InterruptedException | IOException e) {
                e.printStackTrace();
            }
        }).start();
    }

    private static int getThreadCount() {
        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        return threadMXBean.getThreadCount();
    }
}


// Singleton Class for ImageJ instance
class ImageJSingleton {
    private static ImageJ instance;

    private ImageJSingleton() {}

    public static synchronized ImageJ getInstance() {
        if (instance == null) {
            instance = new ImageJ();
            System.out.println("Created new ImageJ instance");
        } else {
            System.out.println("Using existing ImageJ instance");
        }
        return instance;
    }
}