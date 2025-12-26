package com.example.opencvdemo.service;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;
import org.opencv.dnn.Dnn;
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
    String sFaceModel = extractModelToTempFile();

    static {
        Loader.load(opencv_java.class);
    }

    private final FaceRecognizerSF recognizer = FaceRecognizerSF.create(
            sFaceModel, "", Dnn.DNN_BACKEND_OPENCV, Dnn.DNN_TARGET_CPU
    );

    public Mat extractFeatures(Mat origImage, Mat faceImage) {
        Mat targetAligned = new Mat();
        recognizer.alignCrop(origImage, faceImage, targetAligned);
        Mat targetFeatures = new Mat();
        recognizer.feature(targetAligned, targetFeatures);
        return targetFeatures.clone();
    }

    public MatchResult matchFeatures(Mat targetFeatures, Mat queryFeatures) {
        int distanceType = 0;
        double THRESHOLD_COSINE = 0.36;
        double score = recognizer.match(targetFeatures, queryFeatures, distanceType);
        boolean isMatch;
        isMatch = score >= THRESHOLD_COSINE;
        return new MatchResult(score, isMatch);
    }

    private String extractModelToTempFile() {
        try {
            Resource resource = new ClassPathResource("models/face_recognition_sface_2021dec.onnx");
            try {
                return resource.getFile().getAbsolutePath();
            } catch (IOException e) {
                Path tempFile = Files.createTempFile("cvcore_", "_face_recognition.onnx");
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

