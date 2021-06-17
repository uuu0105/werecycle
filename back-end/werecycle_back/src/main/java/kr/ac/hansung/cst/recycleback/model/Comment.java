package kr.ac.hansung.cst.recycleback.model;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import javax.persistence.*;

@Getter
@Setter
@ToString

@Entity
@Table(name="board_comment")
public class Comment {

    @Column(name="post_id")
    private int num;
    private String content;
    @Id
    private String date;
    private String writer;
}