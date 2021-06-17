package kr.ac.hansung.cst.recycleback.dao;

import kr.ac.hansung.cst.recycleback.model.Recycle;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface RecycleDao extends JpaRepository<Recycle, String> {
    List<Recycle> findByCategoryContaining(String category);
    Recycle findByArticle(String article);
}
