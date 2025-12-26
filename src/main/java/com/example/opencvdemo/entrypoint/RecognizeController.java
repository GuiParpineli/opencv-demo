package com.example.opencvdemo.entrypoint;

import com.example.opencvdemo.service.MatchResult;
import com.example.opencvdemo.service.RecognitionService;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/recognize")
public class RecognizeController {
    private final RecognitionService service;

    public RecognizeController(RecognitionService service) {
        this.service = service;
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<MatchResult> compareImages(
            @RequestParam("targetImage") MultipartFile targetImage,
            @RequestParam("queryImage") MultipartFile queryImage
    ) {
        try {
            Mat target = loadFromBytesInMemory(targetImage.getBytes(), targetImage.getOriginalFilename());
            Mat query = loadFromBytesInMemory(queryImage.getBytes(), queryImage.getOriginalFilename());
            Mat queryFeatures = service.extractFeatures(query, query);
            Mat targetFeatures = service.extractFeatures(target, target);
            return ResponseEntity.ok(service.matchFeatures(targetFeatures, queryFeatures));
        } catch (IOException e) {
            return ResponseEntity.badRequest().body(new MatchResult(-1, false));
        }
    }

    public static Mat loadFromBytesInMemory(byte[] bytes, String fileName) {
        if (bytes == null || bytes.length == 0)
            throw new IllegalArgumentException("Arquivo vazio ou inválido: " + fileName);
        MatOfByte mob = new MatOfByte(bytes);
        Mat image = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
        mob.release();
        if (image.empty()) throw new IllegalArgumentException("Não foi possível decodificar a imagem: " + fileName);
        return image;
    }
}
