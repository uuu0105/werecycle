package kr.ac.hansung.cst.recycleback.dao;

import kr.ac.hansung.cst.recycleback.model.Post;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface BoardDao extends JpaRepository<Post, Integer> {
    List<Post> findAllByWriter(String writer);
    List<Post> findByTitleContaining(String title, Pageable paging);
    List<Post> findByPostidLessThanEqual(int postid,Pageable paging);
}
