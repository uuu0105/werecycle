package kr.ac.hansung.cst.recycleback.service;

import kr.ac.hansung.cst.recycleback.dao.FacilityDao;
import kr.ac.hansung.cst.recycleback.dao.FacilityDetailDao;
import kr.ac.hansung.cst.recycleback.model.Facility;
import kr.ac.hansung.cst.recycleback.model.FacilityDetail;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class FacilityService {

    @Autowired
    private FacilityDao facilityDao;

    @Autowired
    private FacilityDetailDao facilityDetailDao;

    public int getTotalNum() {
        return Long.valueOf(Optional.ofNullable(facilityDao.count()).orElse(0L)).intValue();

    }
    public List<Facility> getCurrent(int page, int size){
        Pageable paging = PageRequest.of(page, size, Sort.by("name"));
        return facilityDao.findAll(paging).toList();
    }

    public FacilityDetail getFacilityDetail(String location) {
        return facilityDetailDao.findByName(location);
    }

    public List<Facility> getFacilitiesIn(String search, String word, int page, int size) {
        Pageable paging = PageRequest.of(page, size);
        return facilityDao.findByDistrictContainingOrNameContaining(search, word, paging);
    }

    public int getReqTotalNum(String search, String word) {
        return Long.valueOf(Optional.ofNullable(facilityDao.countByDistrictContainingOrNameContaining(search, word)).orElse(0L)).intValue();
    }
}
