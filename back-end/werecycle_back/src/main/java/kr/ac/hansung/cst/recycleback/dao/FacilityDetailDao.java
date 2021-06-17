package kr.ac.hansung.cst.recycleback.dao;

import kr.ac.hansung.cst.recycleback.model.FacilityDetail;
import org.springframework.data.jpa.repository.JpaRepository;

public interface FacilityDetailDao extends JpaRepository<FacilityDetail, String> {
    FacilityDetail findByName(String name);
}