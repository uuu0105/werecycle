package kr.ac.hansung.cst.recycleback.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import javax.persistence.*;

@Entity
@Getter
@Setter
@ToString
@Table(name="seoulrecyclingfacilities")
public class Facility {

    @Id
    @Column(name="sn")
    private int no;

    @Column(name="city")
    private String district;

    @Column(name="cmpnm")
    private String name;

    @Column(name="cttpc_cn")
    private String tel;

    @JsonIgnore
    @OneToOne(fetch = FetchType.LAZY, optional=false, cascade = CascadeType.ALL)
    @JoinColumn(name="center_name", nullable=false, unique=true)
    private FacilityDetail fd;
}
