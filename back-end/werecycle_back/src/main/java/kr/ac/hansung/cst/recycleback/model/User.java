package kr.ac.hansung.cst.recycleback.model;

import lombok.Getter;
import lombok.Setter;

import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity
@Getter
@Setter
@Table(name="user")
public class User {
    @Id
    private String username;
    private String password;
    private String email;
}
