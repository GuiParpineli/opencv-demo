package com.example.opencvdemo.service;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

@Service
public class RecognitionService {
    private final String sFaceModel = extractModelToTempFile("models/face_recognition_sface_2021dec.onnx");
    private final String yunetPath = extractModelToTempFile("models/face_detection_yunet_2023mar.onnx");
    private static final double THRESHOLD_COSINE = 0.363;
    private static final float DETECT_THRESHOLD = 0.5f;
    private static final float NMS_THRESHOLD = 0.3f;

    static {
        Loader.load(opencv_java.class);
    }

    private final FaceRecognizerSF recognizer = FaceRecognizerSF.create(
            sFaceModel,
            "",
            Dnn.DNN_BACKEND_OPENCV,
            Dnn.DNN_TARGET_CPU
    );

    private final FaceDetectorYN faceDetector = FaceDetectorYN.create(
            yunetPath,
            "",
            new Size(320, 320),
            DETECT_THRESHOLD,
            NMS_THRESHOLD,
            1,
            Dnn.DNN_BACKEND_OPENCV,
            Dnn.DNN_TARGET_CPU
    );

    private Mat extractFeatures(Mat origImage, Mat faceImage) {
        Mat targetAligned = new Mat();
        recognizer.alignCrop(origImage, faceImage, targetAligned);
        Mat targetFeatures = new Mat();
        recognizer.feature(targetAligned, targetFeatures);
        return targetFeatures.clone();
    }

    private Mat infer(Mat image) {
        faceDetector.setInputSize(image.size());
        Mat result = new Mat();
        faceDetector.detect(image, result);
        return result;
    }

    public MatchResult matchFeatures(Mat target, Mat query) {
        Mat targetFaces = infer(target);
        Mat queryFaces = infer(query);

        if (targetFaces.empty() || queryFaces.empty()) {
            return new MatchResult(0.0, false);
        }

        Mat targetFeatures = this.extractFeatures(target, targetFaces.row(0));
        Mat queryFeatures = this.extractFeatures(query, queryFaces.row(0));

        double score = recognizer.match(targetFeatures, queryFeatures, FaceRecognizerSF.FR_COSINE);
        boolean isMatch = score >= THRESHOLD_COSINE;

        return new MatchResult(score, isMatch);
    }

    private String extractModelToTempFile(String path) {
        try {
            Resource resource = new ClassPathResource(path);
            try {
                return resource.getFile().getAbsolutePath();
            } catch (IOException e) {
                Path tempFile = Files.createTempFile("core_", "_face_recognition.onnx");
                tempFile.toFile().deleteOnExit();
                try (InputStream in = resource.getInputStream()) {
                    Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
                }
                return tempFile.toAbsolutePath().toString();
            }
        } catch (IOException e) {
            throw new RuntimeException("Falha ao carregar modelo SFace", e);
        }
    }
}

