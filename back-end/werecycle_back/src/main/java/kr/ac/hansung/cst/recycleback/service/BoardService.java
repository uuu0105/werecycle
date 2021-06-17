package kr.ac.hansung.cst.recycleback.service;

import kr.ac.hansung.cst.recycleback.dao.BoardDao;
import kr.ac.hansung.cst.recycleback.dao.UserDao;
import kr.ac.hansung.cst.recycleback.model.Post;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

@Service
public class BoardService {
    @Autowired
    private BoardDao boardDao;

    @Autowired
    private UserDao userDao;

    public List<Post> getPostPage(int n) {
        Pageable paging = PageRequest.of(n-1, 10,Sort.Direction.DESC, "postid");
        Page<Post> page = boardDao.findAll(paging);
        return page.getContent();
    }

    public int getTotalElements() {
        List<Post> a = boardDao.findAll();
        return a.size();
    }


    public Post getPostById(int id) {
        return boardDao.findById(id).get();
    }

    public List<Post> getPostsByUsername(String username) {
        return boardDao.findAllByWriter(username);
    }

    public List<Post> searchPosts(String word) {
        Pageable paging = PageRequest.of(0,10,Sort.Direction.DESC, "postid");
        return boardDao.findByTitleContaining(word,paging);
    }

    public int createPost(String title, String content, String writer) {
        Post post = new Post();
        post.setTitle(title);
        post.setContent(content);
        post.setWriter(writer);

        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");
        String date = format.format(new Date());
        post.setDate(date);
        Post result=  boardDao.save(post);
        return result.getPostid();
    }

    public Post updatePost(Post post) {
        Post past = boardDao.findById(post.getPostid()).get();
        past.setContent(post.getContent());
        past.setTitle(post.getTitle());
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");
        past.setDate(format.format(new Date()));
        return boardDao.save(past);
    }

    public void deletePost(Post post) {
        boardDao.delete(post);
    }

}
