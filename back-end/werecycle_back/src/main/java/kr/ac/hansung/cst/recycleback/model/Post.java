package kr.ac.hansung.cst.recycleback.model;

import javax.persistence.*;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@Entity
@Table(name="board_post")
public class Post {

    @Id
    @GeneratedValue
    @Column(name="post_id")
    private int postid;

    private String title;
    private String content;
    private String date;
    private String writer;
}
