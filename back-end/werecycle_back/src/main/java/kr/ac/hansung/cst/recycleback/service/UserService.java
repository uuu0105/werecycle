package kr.ac.hansung.cst.recycleback.service;

import kr.ac.hansung.cst.recycleback.dao.UserDao;
import kr.ac.hansung.cst.recycleback.model.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class UserService {

    @Autowired
    private UserDao userDao;

    public String joinUser(User user){
        User compareUser =userDao.findByUsername(user.getUsername());
        if(compareUser != null)
            return "ERROR-USERNAME";
        compareUser = userDao.findByEmail(user.getEmail());
        if(compareUser != null)
            return "ERROR-EMAIL";
        User result = userDao.save(user);
        return result.getUsername();
    }

    public String loginUser(String name, String pw){
        User compareUser =userDao.findByUsername(name);

        if(compareUser == null)
            return "ERROR-NOT_FOUND_USER";
        if(!(compareUser.getPassword().equals(pw)))
            return "ERROR-DOES_NOT_MATCH_PW";
        return compareUser.getUsername();
    }

    public User getInformaion(String name) {
        User result = userDao.findByUsername(name);
        if(result == null)
            return null;
        return result;
    }

    public String changeInform(String name,String pw,String npw1,String npw2) {
        User result = userDao.findByUsername(name);
        if(!result.getPassword().equals(pw)) {
            return "pwError";
        }
        if(!npw1.equals(npw2)) {
            return "npwError";
        }
        result.setPassword(npw1);
        userDao.save(result);
        return "success";
    }
}