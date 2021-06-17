package kr.ac.hansung.cst.recycleback.service;

import kr.ac.hansung.cst.recycleback.dao.CommentDao;
import kr.ac.hansung.cst.recycleback.model.Comment;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.text.SimpleDateFormat;
import java.util.*;

@Service
public class CommentService {
    @Autowired
    private CommentDao commentDao;

    public int createComment(Comment comment){
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");
        String date = format.format(new Date());
        comment.setDate(date);
        Comment result=  commentDao.save(comment);
        return result.getNum();
    }

    public List<Comment> getComments(int num){
        List<Comment> comments = commentDao.findByNum(num);
        return comments;
    }

    public void deleteComment(int id,String date){
        Comment comments = commentDao.findByNumAndDate(id,date);
        commentDao.delete(comments);
    }
}
