package kr.ac.hansung.cst.recycleback.dao;

import kr.ac.hansung.cst.recycleback.model.Article;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.springframework.stereotype.Repository;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Repository
public class CrawlingDao {
    public List<Article> CrawlingPage(int pagenum) {
        try {

            List<Article> artList = new ArrayList<Article>();

            String URL = "http://blog.naver.com/PostList.nhn?from=postList&blogId=mesns&categoryNo=56&parentCategoryNo=56"; //

            if (pagenum != 1)
                URL = URL + "&currentPage=" + pagenum;
            Document html = Jsoup.connect(URL).get();

            Elements contents = html.select("#PostThumbnailAlbumViewArea > ul > li");
            int i=1;
            for (Element thing : contents) {
                Article art = new Article();
                Elements t = thing.select("a");
                art.setNum(i); //
                art.setPageUrl(t.attr("href"));
                Element sont1 = t.get(0);

                art.setImgSrc(sont1.select("img").attr("src"));
                art.setTitle(sont1.select(".title.ell").text());
                art.setDate(sont1.select(".date").text());
                artList.add(art);
                i++;
            }
            return artList;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}