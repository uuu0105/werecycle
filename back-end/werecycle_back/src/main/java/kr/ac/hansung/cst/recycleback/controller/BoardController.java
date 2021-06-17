package kr.ac.hansung.cst.recycleback.controller;

import java.util.List;
import javax.validation.Valid;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;

import kr.ac.hansung.cst.recycleback.model.Comment;
import kr.ac.hansung.cst.recycleback.model.Post;
import kr.ac.hansung.cst.recycleback.service.BoardService;
import kr.ac.hansung.cst.recycleback.service.CommentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import lombok.Getter;
import lombok.Setter;


@CrossOrigin(origins = "http://localhost:3000")
@RestController
@RequestMapping(path = "/weRecycle/boards")
public class BoardController {

    @Autowired
    private BoardService boardService;

    @Autowired
    private CommentService commentService;


    @RequestMapping(path = "/list/{id}", method = RequestMethod.GET)
    public ResponseEntity<?> retrievePostList(@PathVariable int id) {
        int total = boardService.getTotalElements();
        final List<Post> boards = boardService.getPostPage(id);
        if (boards.isEmpty()) {
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        }
        return ResponseEntity.ok(new GetDto(boards,total));
    }

    @GetMapping("/{word}")
    public ResponseEntity<?> searchPost(@PathVariable String word) {
        final List<Post> boards = boardService.searchPosts(word);

        if (boards.isEmpty()) {
            //return new ResponseEntity<>(HttpStatus.NO_CONTENT);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("해당 단어로 검색된 결과가 없습니다. 자유게시글 목록으로 돌아갑니다.");
        }

        return ResponseEntity.ok(boards);
    }

    @RequestMapping(path = "/page/{id}", method = RequestMethod.GET)
    public ResponseEntity<?> retrievePost(@PathVariable int id) {
        final Post post = boardService.getPostById(id);

        if (post == null) {
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        }

        return ResponseEntity.ok(post);
    }

    @RequestMapping(method = RequestMethod.POST)
    public ResponseEntity<?> createPost(@RequestBody @Valid PostDto request) {
        int id = boardService.createPost(request.getTitle(),request.getContent(),request.getWriter());
        return ResponseEntity.status(HttpStatus.CREATED).body(id);
    }

    @RequestMapping(path = "/{id}", method = RequestMethod.PUT)
    public ResponseEntity<?> updatePost(@PathVariable int id, @RequestBody Post request) {

        Post result = boardService.updatePost(request);
        return ResponseEntity.ok().build();

    }

    @RequestMapping(path = "/{id}", method = RequestMethod.DELETE)
    public ResponseEntity<?> deletePost(@PathVariable int id) {
        final Post post = boardService.getPostById(id);
        if(post == null)
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        boardService.deletePost(post);

        return ResponseEntity.noContent().build();
    }

    @RequestMapping(path = "/{id}/comments",method = RequestMethod.POST)
    public ResponseEntity<?> createComment(@PathVariable int id, @RequestBody Comment requests) {
        int n_id = commentService.createComment(requests);
        if(id == n_id)
            return ResponseEntity.ok().build();
        return ResponseEntity.noContent().build();
    }

    @RequestMapping(path = "/{id}/comments",method = RequestMethod.GET)
    public ResponseEntity<?> retrieveComments(@PathVariable int id) {
        final List<Comment> comments = commentService.getComments(id);

        if (comments.isEmpty()) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.ok(comments);
    }

    @RequestMapping(path = "/{id}/comments/{writer}/{date}", method = RequestMethod.DELETE)
    public ResponseEntity<?> deleteComments(@PathVariable int id,@PathVariable String writer,@PathVariable String date) {
        commentService.deleteComment(id,date);
        return ResponseEntity.noContent().build();
    }


    @Getter
    @Setter
    static class PostDto {

        @NotNull(message = "title is required")
        private String title;

        @NotNull(message = "content is required")
        private String content;
        private String writer;
    }

    @Getter
    @Setter
    static class GetDto {
        private List<Post> postList;
        private int totalnum;

        public GetDto(List<Post> postList, int totalnum) {
            this.postList = postList;
            this.totalnum = totalnum;
        }
    }

    @Getter
    @Setter
    static class CommentDto {
        private String writer;
        private String date;
        private int num;
    }
}
