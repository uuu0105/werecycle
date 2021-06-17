package kr.ac.hansung.cst.recycleback.dao;

import kr.ac.hansung.cst.recycleback.model.Facility;
import org.springframework.data.domain.Example;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface FacilityDao extends JpaRepository<Facility, String> {
    Long countByDistrictContainingOrNameContaining(String district, String name);
    List<Facility> findByDistrictContainingOrNameContaining(String district, String name, Pageable paging);
}
