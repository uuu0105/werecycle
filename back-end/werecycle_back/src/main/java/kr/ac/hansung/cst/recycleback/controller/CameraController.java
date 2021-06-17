package kr.ac.hansung.cst.recycleback.controller;

import kr.ac.hansung.cst.recycleback.service.RecycleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

@RestController
@RequestMapping(path = "/weRecycle/camera/ai")
public class CameraController {

    @Autowired
    RecycleService recycleService;

    @RequestMapping(method = RequestMethod.POST)
    public ResponseEntity<?> retrieveImage(@RequestBody String img64encoded) throws IOException {
        if (img64encoded == null) {
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        }
        URL url = new URL("http://172.30.1.60:5000/camera/ai");
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/json; utf-8");
        con.setRequestProperty("Accept", "application/json");
        con.setDoOutput(true);

        String jsonInputString = "{img:" + img64encoded + "}";

        try (OutputStream os = con.getOutputStream()) {
            byte[] input = jsonInputString.getBytes("utf-8");
            os.write(input, 0, input.length);
        }
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(con.getInputStream(), "utf-8"))) {
            StringBuilder response = new StringBuilder();
            String responseLine = null;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
            System.out.println(response.toString());
            return ResponseEntity.ok(recycleService.getArticle(response.toString()));
        }
    }

    @GetMapping("/infor/{article}")
    public ResponseEntity<?> retrieveInfor(@PathVariable String article){
        System.out.println(article);
        return ResponseEntity.ok(recycleService.getInfor(article));
    }
}

