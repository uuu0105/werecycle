package kr.ac.hansung.cst.recycleback.service;

import kr.ac.hansung.cst.recycleback.dao.CrawlingDao;
import kr.ac.hansung.cst.recycleback.model.Article;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class CrawlingService {
    @Autowired
    private CrawlingDao offerDao;

    public List<Article> getIssuePages(int pagenum) {
        return offerDao.CrawlingPage(pagenum);
    }
}
