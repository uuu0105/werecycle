package kr.ac.hansung.cst.recycleback.controller;

import kr.ac.hansung.cst.recycleback.model.Facility;
import kr.ac.hansung.cst.recycleback.model.FacilityDetail;
import kr.ac.hansung.cst.recycleback.service.FacilityService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.List;

@RestController
@RequestMapping(path = "/weRecycle/facilities")
public class FacilityController {

    @Autowired
    private FacilityService facilityService;

    @RequestMapping(method = RequestMethod.GET)
    public ResponseEntity<List<Facility>> retrieveFacilities(
            @RequestParam(value="page",defaultValue="0") int page,
            @RequestParam(value="size",defaultValue="10") int size) {

            List<Facility> facilities = facilityService.getCurrent(page,size);
            int f_totalNum = facilityService.getTotalNum();

            if(facilities.isEmpty()) {
                return new ResponseEntity<>(HttpStatus.NO_CONTENT);
            }
            HttpHeaders httpHeaders = new HttpHeaders();
            httpHeaders.add("totalDNum", Integer.toString(f_totalNum));

            return new ResponseEntity<List<Facility>>(facilities, httpHeaders, HttpStatus.OK);
    }

    @RequestMapping(path="/search" , method = RequestMethod.GET)
    public ResponseEntity<List<Facility>> retrieveReqFacilities(
            @RequestParam(value="search",required=false) String search,
            @RequestParam(value="word",required=false) String word,
            @RequestParam(value="page",defaultValue="0") int page,
            @RequestParam(value="size",defaultValue="10") int size) {

        String searchStr =  search==null ? "none" : search;
        String wordStr = word==null ? "none" : word;

        List<Facility> reqFacilities = facilityService.getFacilitiesIn(searchStr, wordStr, page, size);
        int req_totalNum = facilityService.getReqTotalNum(searchStr, wordStr);

        if( reqFacilities.isEmpty()) {
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        }

        HttpHeaders httpHeaders = new HttpHeaders();
        httpHeaders.add("totalDNum", Integer.toString(req_totalNum));

        return new ResponseEntity<List<Facility>>(reqFacilities, httpHeaders, HttpStatus.OK);

    }

    @RequestMapping(path = "/detail", method = RequestMethod.GET)
    public ResponseEntity<FacilityDetail> retrieveFacilities(
            @RequestParam(value="n",required=true) String n) throws UnsupportedEncodingException {
        String decodeResult = URLDecoder.decode(n, "UTF-8");
        FacilityDetail fd = facilityService.getFacilityDetail(n);
        if (fd != null) {
            return new ResponseEntity<FacilityDetail>(fd, HttpStatus.OK);
        }
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }

}
