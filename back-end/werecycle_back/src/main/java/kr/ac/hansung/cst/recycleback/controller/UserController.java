package kr.ac.hansung.cst.recycleback.controller;

import kr.ac.hansung.cst.recycleback.model.Post;
import kr.ac.hansung.cst.recycleback.model.User;
import kr.ac.hansung.cst.recycleback.service.BoardService;
import kr.ac.hansung.cst.recycleback.service.UserService;
import lombok.Getter;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@CrossOrigin(origins = "http://localhost:3000")
@RestController
@RequestMapping(path = "/weRecycle/users")
public class UserController {

    @Autowired
    private UserService userService;

    @Autowired
    private BoardService boardService;

    @PostMapping("/join")
    public ResponseEntity<?> join(@RequestBody User user) {
        String result = userService.joinUser(user);
        if(result.equals("ERROR-USERNAME")) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("사용중인 아이디 입니다.");
        }
        else if(result.equals("ERROR-EMAIL")) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("사용중인 이메일 입니다.");
        }
        return ResponseEntity.ok().build();
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody UserDto user){
        String result = userService.loginUser(user.getUsername(), user.getPassword());
        if(result.equals("ERROR-NOT_FOUND_USER")) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(result);
        }
        else if(result.equals("ERROR-DOES_NOT_MATCH_PW")) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(result);
        }
        return ResponseEntity.ok(result);
    }

    @GetMapping("/mypage/{name}")
    public ResponseEntity<?> myInform(@PathVariable String name){
        User foundUser = userService.getInformaion(name);
        return ResponseEntity.ok(foundUser);
    }

    @PutMapping("/mypage")
    public ResponseEntity<?> changeMyInform(@RequestBody UserDto user){
        String result = userService.changeInform(user.getUsername(),user.getPassword(),user.getNewPassword1(),user.getNewPassword2());
        if(result.equals("pwError"))
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("현재 비밀번호가 정확하지 않습니다.");
        if(result.equals("npwError"))
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("변경될 비밀번호가 동일하지 않습니다.");
        return ResponseEntity.ok("성공적으로 변경하였습니다.");
    }

    @GetMapping("/mypage/{name}/posts")
    public ResponseEntity<?> myPosts(@PathVariable String name){
        List<Post> posts = boardService.getPostsByUsername(name);
        if (posts.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(posts);
    }

    @Getter
    @Setter
    static class UserDto {
        private String username;
        private String password;
        private String newPassword1;
        private String newPassword2;
    }

}

