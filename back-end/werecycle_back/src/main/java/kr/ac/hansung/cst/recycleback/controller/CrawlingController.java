package kr.ac.hansung.cst.recycleback.controller;

import kr.ac.hansung.cst.recycleback.model.Article;
import kr.ac.hansung.cst.recycleback.service.CrawlingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@CrossOrigin(origins = "http://localhost:3000")
@RestController
@RequestMapping(path = "/weRecycle/issues")
public class CrawlingController {
    @Autowired
    private CrawlingService offerService;

    @RequestMapping(path = "/{pagenum}", method = RequestMethod.GET)
    public ResponseEntity<?> showOffers(@PathVariable("pagenum") int pagenum) {
        List<Article> articles=offerService.getIssuePages(pagenum);
        return ResponseEntity.ok(articles);
    }
}
