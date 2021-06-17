package kr.ac.hansung.cst.recycleback.service;

import kr.ac.hansung.cst.recycleback.dao.RecycleDao;
import kr.ac.hansung.cst.recycleback.model.Recycle;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.ArrayList;
import java.util.List;

@Service
public class RecycleService {

    @Autowired
    private RecycleDao recycleDao;

    public Recycle createRecycle(Recycle recycle){
        return recycleDao.save(recycle);
    }

    public List<String> getArticle(String category){
        List<Recycle> recycles = recycleDao.findByCategoryContaining(category);
        List<String> articles = new ArrayList<>();
        for(Recycle recycle: recycles){
            articles.add(recycle.getArticle());
        }
        return articles;
    }

    public String getInfor(String article){
        Recycle recycle = recycleDao.findByArticle(article);
        return recycle.getInfor();
    }
}
