package kr.ac.hansung.cst.recycleback.model;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Table;
import javax.persistence.Id;

@Entity
@Getter
@Setter
@ToString
@Table(name="detail_location")
public class FacilityDetail {

    @Id
    @Column(name="center_name")
    private String name;

    private String address_detail;

    private String latitude;
    private String longitude;

    private String tel;

    private String opentime;
    private String closetime;

    private String opentime_holiday;
    private String closetime_holiday;

    private String off_info;
    private String site_url;
}
