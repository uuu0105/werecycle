package kr.ac.hansung.cst.recycleback.dao;

import kr.ac.hansung.cst.recycleback.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface UserDao extends JpaRepository<User,String> {
    User findByUsername(String username);
    User findByEmail(String email);
}
