package kr.ac.hansung.cst.recycleback.model;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import javax.persistence.*;
import javax.persistence.Entity;
import java.util.List;

@Entity
@Getter
@Setter
@ToString
@Table(name="recycle")
public class Recycle {
    @Id
    private String article;
    private String category;
    private String infor;
}