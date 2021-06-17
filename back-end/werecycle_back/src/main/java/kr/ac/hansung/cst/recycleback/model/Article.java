package kr.ac.hansung.cst.recycleback.model;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import org.jsoup.nodes.Element;

import javax.persistence.Entity;
import javax.persistence.Table;

@Getter
@Setter
@ToString
@NoArgsConstructor
public class Article {

    int num;
    String pageUrl;
    String imgSrc;
    String title;
    String date;
}
