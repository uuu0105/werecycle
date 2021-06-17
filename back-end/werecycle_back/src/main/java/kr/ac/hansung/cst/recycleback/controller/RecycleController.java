package kr.ac.hansung.cst.recycleback.controller;

import kr.ac.hansung.cst.recycleback.model.Recycle;
import kr.ac.hansung.cst.recycleback.service.RecycleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins = "http://localhost:3000")
@RestController
@RequestMapping(path = "weRecycle/recycles")
public class RecycleController {

    @Autowired
    RecycleService recycleService;

    @GetMapping("/article/{category}")
    public ResponseEntity<?> retrieveArticle(@PathVariable String category){
        System.out.println(category);
        return ResponseEntity.ok(recycleService.getArticle(category));
    }

    @GetMapping("/infor/{article}")
    public ResponseEntity<?> retrieveInfor(@PathVariable String article){
        System.out.println(article);
        return ResponseEntity.ok(recycleService.getInfor(article));
    }
}
