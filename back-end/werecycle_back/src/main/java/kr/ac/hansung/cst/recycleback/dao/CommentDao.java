package kr.ac.hansung.cst.recycleback.dao;

import kr.ac.hansung.cst.recycleback.model.Comment;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface CommentDao extends JpaRepository<Comment, String> {
    List<Comment> findByNum(int num);
    Comment findByNumAndDate(int num, String date);
}