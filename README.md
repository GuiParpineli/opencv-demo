# OpenCV Demo Application

A Spring Boot application that demonstrates face recognition capabilities using OpenCV and the SFace model.

## Overview

This project provides a REST API to compare two images and determine if they contain the same person. It leverages the
OpenCV library via the Bytedeco wrapper and uses the SFace (Face Recognition) model.

## Tech Stack

- **Language:** Java 25
- **Framework:** Spring Boot 4.0.1
- **Package Manager:** Gradle (Kotlin DSL)
- **Computer Vision:** OpenCV (via `org.bytedeco:opencv-platform`)
- **API Documentation:** SpringDoc OpenAPI (Swagger UI)

## Requirements

- **JDK:** Java 25
- **Operating System:** Compatible with OpenCV (Windows, macOS, Linux)

## Setup and Run

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd opencv-demo
   ```

2. **Build the project:**
   ```bash
   ./gradlew build
   ```

3. **Run the application:**
   ```bash
   ./gradlew bootRun
   ```

The application will start on `http://localhost:8080` by default.

## API Documentation

Once the application is running, you can access the Swagger UI at:
`http://localhost:8080/swagger-ui/index.html`

## API Endpoints

### Recognize (Compare Images)

- **URL:** `/recognize`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
    - `targetImage` (File): The target image.
    - `queryImage` (File): The image to compare against the target.
- **Success Response:**
    - **Code:** 200 OK
    - **Content:** `{ "score": 0.85, "match": true }`

## Available Scripts

- `./gradlew bootRun`: Runs the Spring Boot application.
- `./gradlew build`: Compiles and packages the application into a JAR file.
- `./gradlew test`: Executes unit tests.
- `./gradlew clean`: Removes the build directory.

## Project Structure

```text
opencv-demo/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/example/opencvdemo/
│   │   │       ├── OpencvDemoApplication.java  # Entry point
│   │   │       ├── entrypoint/                 # REST Controllers
│   │   │       │   └── RecognizeController.java
│   │   │       └── service/                    # Business Logic
│   │   │           ├── RecognitionService.java # OpenCV logic
│   │   │           └── MatchResult.java        # Data model
│   │   └── resources/
│   │       ├── application.yaml                # Configuration
│   │       └── models/                         # ML Models
│   │           └── face_recognition_sface_2021dec.onnx
│   └── test/                                   # Unit and Integration tests
├── build.gradle.kts                            # Gradle dependencies
└── README.md                                   # Project documentation
```

## Tests

To run the tests, execute:

```bash
./gradlew test
```